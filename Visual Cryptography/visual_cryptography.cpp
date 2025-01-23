#include <iostream>
#include <string>
#include <cstdlib>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <opencv2/opencv.hpp>

using namespace std;
namespace po = boost::program_options;

// Generate the ownership share
cv::Mat visual_cryptography(const cv::Mat& host, const cv::Mat& secret, const int& key) {
    cv::Size binary_size(host.rows / 2, host.cols / 2);
    cv::Mat binary_mat(binary_size, CV_8UC1);
    cv::Mat ownership_share(host.size(), CV_8UC1, cv::Scalar(0));

    // Resize and threshold the secret image
    cv::Mat resized_secret;
    cv::resize(secret, resized_secret, binary_size);
    cv::cvtColor(resized_secret, resized_secret, cv::COLOR_BGR2GRAY);
    cv::threshold(resized_secret, resized_secret, 128, 255, cv::THRESH_BINARY);

    // Initialize random seed
    srand(key);

    // Generate binary_mat
    for (int row_idx = 0; row_idx < binary_size.height; row_idx++) {
        for (int col_idx = 0; col_idx < binary_size.width; col_idx++) {
            binary_mat.at<uchar>(row_idx, col_idx) = rand() % 2;
        }
    }

    // Generate ownership share
    for (int row_idx = 0; row_idx < binary_size.height; row_idx++) {
        for (int col_idx = 0; col_idx < binary_size.width; col_idx++) {
            int base_row = row_idx * 2, base_col = col_idx * 2;
            uchar bit = binary_mat.at<uchar>(row_idx, col_idx);

            if (bit == 0) {
                ownership_share.at<uchar>(base_row, base_col) = 0;
                ownership_share.at<uchar>(base_row + 1, base_col) = 255;
                ownership_share.at<uchar>(base_row, base_col + 1) = 0;
                ownership_share.at<uchar>(base_row + 1, base_col + 1) = 255;
            } else {
                ownership_share.at<uchar>(base_row, base_col) = 255;
                ownership_share.at<uchar>(base_row + 1, base_col) = 0;
                ownership_share.at<uchar>(base_row, base_col + 1) = 255;
                ownership_share.at<uchar>(base_row + 1, base_col + 1) = 0;
            }
        }
    }

    // Create watermark from the ownership share and secret image
    cv::Mat watermark(host.size(), CV_8UC1, cv::Scalar(0));
    for (int row_idx = 0; row_idx < binary_size.height; row_idx++) {
        for (int col_idx = 0; col_idx < binary_size.width; col_idx++) {
            int base_row = row_idx * 2, base_col = col_idx * 2;

            if (resized_secret.at<uchar>(row_idx, col_idx) == 255) {
                watermark.at<uchar>(base_row, base_col) = ownership_share.at<uchar>(base_row, base_col);
                watermark.at<uchar>(base_row + 1, base_col) = ownership_share.at<uchar>(base_row + 1, base_col);
                watermark.at<uchar>(base_row, base_col + 1) = ownership_share.at<uchar>(base_row, base_col + 1);
                watermark.at<uchar>(base_row + 1, base_col + 1) = ownership_share.at<uchar>(base_row + 1, base_col + 1);
            } else {
                watermark.at<uchar>(base_row, base_col) = ownership_share.at<uchar>(base_row + 1, base_col);
                watermark.at<uchar>(base_row + 1, base_col) = ownership_share.at<uchar>(base_row, base_col);
                watermark.at<uchar>(base_row, base_col + 1) = ownership_share.at<uchar>(base_row + 1, base_col + 1);
                watermark.at<uchar>(base_row + 1, base_col + 1) = ownership_share.at<uchar>(base_row, base_col + 1);
            }
        }
    }

    return watermark;
}

// Decrypt to reveal the original secret image
cv::Mat decrypt(const cv::Mat& ownership_share, const cv::Mat& host, const int& key) {
    cv::Size binary_size(host.rows / 2, host.cols / 2);
    cv::Mat binary_mat(binary_size, CV_8UC1);

    srand(key);

    // Regenerate the binary_mat using the same random sequence
    for (int row_idx = 0; row_idx < binary_size.height; row_idx++) {
        for (int col_idx = 0; col_idx < binary_size.width; col_idx++) {
            binary_mat.at<uchar>(row_idx, col_idx) = rand() % 2;
        }
    }

    // Recreate the binary master share
    cv::Mat binary_master_share(host.size(), CV_8UC1, cv::Scalar(0));
    for (int row_idx = 0; row_idx < binary_mat.rows; row_idx++) {
        for (int col_idx = 0; col_idx < binary_mat.cols; col_idx++) {
            int base_row = row_idx * 2, base_col = col_idx * 2;

            uchar bit = binary_mat.at<uchar>(row_idx, col_idx);
            if (bit == 0) {
                binary_master_share.at<uchar>(base_row, base_col) = 0;
                binary_master_share.at<uchar>(base_row + 1, base_col) = 255;
                binary_master_share.at<uchar>(base_row, base_col + 1) = 0;
                binary_master_share.at<uchar>(base_row + 1, base_col + 1) = 255;
            } else {
                binary_master_share.at<uchar>(base_row, base_col) = 255;
                binary_master_share.at<uchar>(base_row + 1, base_col) = 0;
                binary_master_share.at<uchar>(base_row, base_col + 1) = 255;
                binary_master_share.at<uchar>(base_row + 1, base_col + 1) = 0;
            }
        }
    }

    // Reveal the watermark using the ownership share and the master share
    cv::Mat revealed_watermark(host.size(), CV_8UC1, cv::Scalar(0));
    for (int row_idx = 0; row_idx < host.rows; row_idx++) {
        for (int col_idx = 0; col_idx < host.cols; col_idx++) {
            uchar ownership_pixel = ownership_share.at<uchar>(row_idx, col_idx);
            uchar master_pixel = binary_master_share.at<uchar>(row_idx, col_idx);

            revealed_watermark.at<uchar>(row_idx, col_idx) =
                (ownership_pixel == master_pixel) ? binary_master_share.at<uchar>(row_idx, col_idx) : 0;
        }
    }

    return revealed_watermark;
}

int main(int argc, char* argv[]) {
    bool verbose = false;
    string host_filename, secret_filename;
    int key;

    // Define program options
    po::options_description options("Options");
    options.add_options()
        ("help,h", "Display this message")
        ("verbose,v", po::bool_switch(&verbose), "Enable verbose output")
        ("host-image,i", po::value<string>(&host_filename)->required(), "Host image file")
        ("secret-image,s", po::value<string>(&secret_filename)->required(), "Secret image file")
        ("key,k", po::value<int>(&key)->required(), "Private key");

    po::variables_map vm;
    try {
        // Parse command line arguments with options_description
        po::store(po::parse_command_line(argc, argv, options), vm);

        if (vm.count("help")) {
            cout << options << endl;
            return EXIT_SUCCESS;
        }

        // Notify will throw an error if required options are missing
        po::notify(vm);
    } catch (const po::error& ex) {
        cerr << "*** ERROR *** " << ex.what() << endl;
        return EXIT_FAILURE;
    }

    if (!boost::filesystem::exists(host_filename) || !boost::filesystem::exists(secret_filename)) {
        cerr << "*** ERROR *** Provided files do not exist." << endl;
        return EXIT_FAILURE;
    }

    // Load the host and secret images
    cv::Mat host = cv::imread(host_filename, cv::IMREAD_COLOR);
    cv::Mat secret = cv::imread(secret_filename, cv::IMREAD_COLOR);

    if (host.empty() || secret.empty()) {
        cerr << "*** ERROR *** Failed to load images." << endl;
        return EXIT_FAILURE;
    }

    // Generate watermark (ownership share)
    cv::Mat ownership_share = visual_cryptography(host, secret, key);

    // Decrypt the watermark
    cv::Mat revealed_watermark = decrypt(ownership_share, host, key);

    // Display results
    cv::imshow("Host Image", host);
    cv::imshow("Secret Image", secret);
    cv::imshow("Ownership Share", ownership_share);
    cv::imshow("Revealed Watermark", revealed_watermark);
    cv::waitKey(0);

    return EXIT_SUCCESS;
}
