#include <cmath>
#include <iostream>
#include <chrono>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <mpi.h>
//Mohd Hakimie 1171302044
//to be used with Smalltalk

const int x[3][3] = { {-1,0,1}, {-2,0,2}, {-1,0,1} };
const int y[3][3] = { {-1,-2,-1}, {0,0,0}, {1,2,1} };

//for MPI uses
int flag, signal;
int num_proc;
int my_rank;
MPI_Request request;
MPI_Comm parentcomm;

using namespace cv;
using namespace std;


cv::Mat sobelTransform(const cv::Mat& inputImage, const cv::Mat& initialImage) {

    cv::Mat filteredImage = cv::Mat::zeros(inputImage.size(), inputImage.type());

    for (int j = 0; j < inputImage.rows; ++j) {
        for (int i = 0; i < (inputImage.cols - 2); ++i) {
            int xValOfPixel;
            int yValOfPixel;
            if (j < inputImage.rows - 2) {
                // Calculate X gradient of pixel
                xValOfPixel =
                    (x[0][0] * (int)inputImage.at<uchar>(j, i)) + (x[0][2] * (int)inputImage.at<uchar>(j + 2, i)) +
                    (x[1][0] * (int)inputImage.at<uchar>(j, i + 1)) + (x[1][2] * (int)inputImage.at<uchar>(j + 2, i + 1)) +
                    (x[2][0] * (int)inputImage.at<uchar>(j, i + 2)) + (x[2][2] * (int)inputImage.at<uchar>(j + 2, i + 2));
                // Calculate Y gradient of pixel
                yValOfPixel =
                    (y[0][0] * (int)inputImage.at<uchar>(j, i)) + (y[0][1] * (int)inputImage.at<uchar>(j + 1, i)) + (y[0][2] * (int)inputImage.at<uchar>(j + 2, i)) +
                    (y[2][0] * (int)inputImage.at<uchar>(j, i + 2)) + (y[2][1] * (int)inputImage.at<uchar>(j + 1, i + 2)) + (y[2][2] * (int)inputImage.at<uchar>(j + 2, i + 2));

                // Calculate magnitude (absolute aproximation)
                int sum = std::clamp(std::abs(xValOfPixel) + std::abs(yValOfPixel), 0, 255);
                filteredImage.at<uchar>(j, i) = (uchar)sum;
            }
            else if (i < (inputImage.cols - 2) && my_rank < num_proc - 1) {
                // Calculate X gradient of pixel
                int k = j * (my_rank + 1);
                xValOfPixel =
                    (x[0][0] * (int)initialImage.at<uchar>(k, i)) + (x[0][2] * (int)initialImage.at<uchar>(k + 2, i)) +
                    (x[1][0] * (int)initialImage.at<uchar>(k, i + 1)) + (x[1][2] * (int)initialImage.at<uchar>(k + 2, i + 1)) +
                    (x[2][0] * (int)initialImage.at<uchar>(k, i + 2)) + (x[2][2] * (int)initialImage.at<uchar>(k + 2, i + 2));
                // Calculate Y gradient of pixel
                yValOfPixel =
                    (y[0][0] * (int)initialImage.at<uchar>(k, i)) + (y[0][1] * (int)initialImage.at<uchar>(k + 1, i)) + (y[0][2] * (int)initialImage.at<uchar>(k + 2, i)) +
                    (y[2][0] * (int)initialImage.at<uchar>(k, i + 2)) + (y[2][1] * (int)initialImage.at<uchar>(k + 1, i + 2)) + (y[2][2] * (int)initialImage.at<uchar>(k + 2, i + 2));

                // Calculate magnitude (absolute aproximation)
                int sum = std::clamp(std::abs(xValOfPixel) + std::abs(yValOfPixel), 0, 255);
                filteredImage.at<uchar>(j, i) = (uchar)sum;
            }

        }
    }
    return filteredImage;
}

int main(int argc, char** argv) {

    char buf[256];
    Mat initialImage;
    Mat finalImage;


    /* Initialize the infrastructure necessary for communication */
    MPI_Init(&argc, &argv);

    /* Identify this process */
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    /* Find out how many total processes are active */
    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);

    /* Get parent communicator*/
    MPI_Comm_get_parent(&parentcomm);

    /* Receive imagename */
    MPI_Bcast(&buf, sizeof(buf), MPI_CHAR, 0, parentcomm);


    if (my_rank == 0) {
        //Load image in greyscale
        initialImage = imread(buf, cv::IMREAD_GRAYSCALE);
        cv::namedWindow("Sobel Filter", cv::WINDOW_NORMAL);
        cv::imshow("Sobel Filter", initialImage);
        int tempRow = initialImage.rows;
        while (tempRow % num_proc != 0) {
            tempRow--;
        }
        if (tempRow != initialImage.rows) {
            Mat roi(initialImage, Rect(0, 0, initialImage.cols, tempRow));
            initialImage = roi.clone();
        }
    }

    MPI_Ibcast(&signal, 1, MPI_INT, 0, parentcomm, &request);

    while (signal != 1) {

        /* Receive the next instruction from Smalltalk */
        cv::waitKey(1);

        /* Apply Sobel */
        if (flag == 1 && signal == 2) {

            //declare variable for image
            Mat temp;
            size_t sizeInBytes;
            unsigned long long smallCount;
            int imageRow, imageColumn, imageType;
            double startTime, endTime;

            //assign image properties
            if (my_rank == 0) {

                startTime = MPI_Wtime();
                sizeInBytes = initialImage.total() * initialImage.elemSize();
                imageRow = initialImage.rows;
                imageColumn = initialImage.cols;
                imageType = initialImage.type();
            }

            //pass image properties
            MPI_Bcast(&imageRow, 1, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast(&imageColumn, 1, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast(&imageType, 1, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast(&sizeInBytes, 1, MPI_LONG, 0, MPI_COMM_WORLD);

            //initialize image base on the properties
            finalImage = Mat::zeros(imageRow, imageColumn, imageType);

            if (my_rank != 0) {
                initialImage = Mat::zeros(imageRow, imageColumn, imageType);
            }

            //pass the image
            MPI_Bcast(initialImage.data, sizeInBytes, MPI_BYTE, 0, MPI_COMM_WORLD);

            //calculate the image size for each processor
            smallCount = (initialImage.rows / num_proc) * initialImage.cols;

            //initialize image
            temp = Mat::zeros(imageRow / num_proc, imageColumn, initialImage.type());

            //scatter the image
            MPI_Scatter(initialImage.data, smallCount, MPI_BYTE, temp.data, smallCount, MPI_BYTE, 0, MPI_COMM_WORLD);
            
            //apply the function
            temp = sobelTransform(temp, initialImage);

            //gather back the image
            MPI_Gather(temp.data, smallCount, MPI_BYTE, finalImage.data, smallCount, MPI_BYTE, 0, MPI_COMM_WORLD);

            //send the execution time to Smalltalk
            if (my_rank == 0) {
                endTime = MPI_Wtime();
                double totalTime = endTime - startTime;
                MPI_Send(&totalTime, 1, MPI_DOUBLE, 0, 0, parentcomm);
                cv::imshow("Sobel Filter", finalImage);
            }

            //ready to receive the next instruction
            MPI_Ibcast(&signal, 1, MPI_INT, 0, parentcomm, &request);
            flag = 0;
        }
        /* Undo Sobel */
        else if (flag == 1 && signal == 3) {

            if (my_rank == 0) {
                finalImage = initialImage.clone();
                cv::imshow("Sobel Filter", finalImage);
            }
            MPI_Ibcast(&signal, 1, MPI_INT, 0, parentcomm, &request);
            flag = 0;
        }

        //increase brightness
        else if (flag == 1 && signal == 4) {
            if (my_rank == 0) {
                if (finalImage.data != 0) {
                    finalImage.convertTo(finalImage, -1, 1, 25);
                    cv::imshow("Sobel Filter", finalImage);
                }
                else {
                    initialImage.convertTo(initialImage, -1, 1, 25);
                    cv::imshow("Sobel Filter", initialImage);
                }
            }
            //ready to receive the next instruction
            MPI_Ibcast(&signal, 1, MPI_INT, 0, parentcomm, &request);
            flag = 0;
        }
        // decrease brightness
        else if (flag == 1 && signal == 5) {
            if (my_rank == 0) {
                if (finalImage.data != 0) {
                    finalImage.convertTo(finalImage, -1, 1, -25);
                    cv::imshow("Sobel Filter", finalImage);
                }
                else {
                    initialImage.convertTo(initialImage, -1, 1, -25);
                    cv::imshow("Sobel Filter", initialImage);
                }
            }
            //ready to receive the next instruction
            MPI_Ibcast(&signal, 1, MPI_INT, 0, parentcomm, &request);
            flag = 0;
        }
        //save image
        else if (flag == 1 && signal == 6) {
            if (my_rank == 0) {
                string fileName = buf;
                fileName = fileName.substr(0, fileName.find_last_of(".")) + "Filtered" + fileName.substr(fileName.find_last_of("."));
                imwrite(fileName, finalImage);
            }
            //ready to receive the next instruction
            MPI_Ibcast(&signal, 1, MPI_INT, 0, parentcomm, &request);
            flag = 0;
        }
        else {
            MPI_Test(&request, &flag, MPI_STATUS_IGNORE);
        }
    }

    destroyAllWindows();
    MPI_Comm_disconnect(&parentcomm);
    MPI_Finalize();
}