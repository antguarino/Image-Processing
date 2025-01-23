#include <iostream>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;

namespace po = boost::program_options;

int main(int argc, char* argv[]) {
    bool verbose = false;
    string src_filename = "";
    string output_filename = "";

    po::options_description options("Options");
    options.add_options()
        ("help,h", "display this message")
        ("verbose,v", po::bool_switch(&verbose), "display verbose output")
        ("source-filename,i", po::value<string>(&src_filename), "source filename")
        ("output-filename,o", po::value<string>(&output_filename), "output filename for magnitude image");

    po::positional_options_description positional_options;
    positional_options.add("source-filename", -1);

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv)
                .options(options)
                .positional(positional_options)
                .run(), vm);
    po::notify(vm);

    if (vm.count("help")) {
        cout << "Usage: " << argv[0] << " [options] source-filename" << endl;
        cout << options << endl;
        return EXIT_SUCCESS;
    }

    if (!boost::filesystem::exists(src_filename)) {
        cerr << "Provided source file does not exist" << endl;
        return EXIT_FAILURE;
    }

    cv::Mat src = cv::imread(src_filename, cv::IMREAD_COLOR);
    if (src.empty()) {
        cerr << "Could not read the image!" << endl;
        return EXIT_FAILURE;
    }

    cv::Mat gray, dst_x, dst_y;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);

    cv::Sobel(gray, dst_x, CV_16S, 1, 0, 3);
    cv::Sobel(gray, dst_y, CV_16S, 0, 1, 3);

    cv::Mat float_dst_x, float_dst_y;
    dst_x.convertTo(float_dst_x, CV_32F);
    dst_y.convertTo(float_dst_y, CV_32F);

    cv::Mat magnitude, direction;
    cv::magnitude(float_dst_x, float_dst_y, magnitude);
    cv::phase(float_dst_x, float_dst_y, direction, true);

    cv::normalize(magnitude, magnitude, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::normalize(dst_x, dst_x, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::normalize(dst_y, dst_y, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::normalize(direction, direction, 0, 255, cv::NORM_MINMAX, CV_8U);

    if (verbose) {
        cv::imshow("Original Color", src);
        cv::imshow("Sobel Gx", dst_x);
        cv::imshow("Sobel Gy", dst_y);
        cv::imshow("Magnitude", magnitude);
        cv::imshow("Direction", direction);
        cv::waitKey(0);
    }

    if (!output_filename.empty()) {
        cv::imwrite(output_filename, magnitude);
    }

    return EXIT_SUCCESS;
}
