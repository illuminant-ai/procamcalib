/*
Copyright (c) 2012, Daniel Moreno and Gabriel Taubin
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the Brown University nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL DANIEL MORENO AND GABRIEL TAUBIN BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "CalibrationData.hpp"

#include "Application.hpp"

#include <QFileInfo>

#include <ctime>
#include <regex>
#include <iostream>
#include <opencv2/calib3d.hpp>

CalibrationData::CalibrationData() :
    cam_K(), cam_kc(),
    proj_K(), proj_kc(),
    R(), T(), E(), F(),
    // H1(), H2(),
    cam_error(0.0), proj_error(0.0), stereo_error(0.0),
    cam_height(0.0), cam_width(0.0), proj_height(0.0), proj_width(0.0),
    filename()
{
}

CalibrationData::~CalibrationData()
{
}

void CalibrationData::clear(void)
{
    cam_K = cv::Mat();
    cam_kc = cv::Mat();
    proj_K = cv::Mat();
    proj_kc = cv::Mat();
    R = cv::Mat();
    T = cv::Mat();
    E = cv::Mat();
    F = cv::Mat();
    // H1 = cv::Mat();
    // H2 = cv::Mat();
    filename = QString();

    cam_error = 0.0;
    proj_error = 0.0;
    stereo_error = 0.0;

    cam_width = 0;
    cam_height = 0;
    proj_width = 0;
    proj_height = 0;

    cam_per_view_errors = cv::Mat();
    proj_per_view_errors = cv::Mat();
    stereo_per_view_errors = cv::Mat();
}

bool CalibrationData::is_valid(void) const
{
    return (cam_K.data && cam_kc.data && proj_K.data && proj_kc.data && R.data && T.data && E.data && F.data/* && H1.data && H2.data*/);
}

bool CalibrationData::load_calibration(QString const& filename)
{
    QFileInfo info(filename);
    QString type = info.suffix();

    if (type=="yml") {return load_calibration_yml(filename);}

    return false;
}

bool CalibrationData::save_calibration(QString const& filename)
{
    QFileInfo info(filename);
    QString type = info.suffix();

    if (type=="yml") {return save_calibration_yml(filename);}
    if (type=="m"  ) {return save_calibration_matlab(filename);}

    return false;
}

bool CalibrationData::load_calibration_yml(QString const& filename)
{
    cv::FileStorage fs(filename.toStdString(), cv::FileStorage::READ);
    if (!fs.isOpened())
    {
        return false;
    }

    fs["cam_K"] >> cam_K;
    fs["cam_kc"] >> cam_kc;
    fs["proj_K"] >> proj_K;
    fs["proj_kc"] >> proj_kc;
    fs["R"] >> R;
    fs["T"] >> T;
    fs["E"] >> E;
    fs["F"] >> F;
    // fs["H1"] >> H1;
    // fs["H2"] >> H2;

    fs["cam_error"] >> cam_error;
    fs["proj_error"] >> proj_error;
    fs["stereo_error"] >> stereo_error;

    fs.release();

    this->filename = filename;

    return true;
}

bool CalibrationData::save_calibration_yml(QString const& filename)
{
    cv::FileStorage fs(filename.toStdString(), cv::FileStorage::WRITE);
    if (!fs.isOpened())
    {
        return false;
    }

    fs << "cam_K" << cam_K << "cam_kc" << cam_kc
       << "proj_K" << proj_K << "proj_kc" << proj_kc
       << "R" << R << "T" << T << "E" << E << "F" << F
    //    << "H1" << H1 << "H2" << H2
       << "cam_error" << cam_error
       << "proj_error" << proj_error
       << "stereo_error" << stereo_error
       ;
    fs.release();

    this->filename = filename;

    return true;
}

static cv::Mat convert_transform_from_opencv_to_unity(cv::Mat transform)
{
    cv::Mat out = transform.clone();
    out.at<double>(0, 1) = -transform.at<double>(0, 1);
    out.at<double>(1, 0) = -transform.at<double>(1, 0);
    out.at<double>(1, 2) = -transform.at<double>(1, 2);
    out.at<double>(1, 3) = -transform.at<double>(1, 3);
    out.at<double>(2, 1) = -transform.at<double>(2, 1);
    out.at<double>(3, 0) = -transform.at<double>(3, 0);
    out.at<double>(3, 2) = -transform.at<double>(3, 2);
    return out;
}

static cv::Mat convert_transform_from_unity_to_scene(cv::Mat transform)
{
    cv::Mat out = transform.clone();
    cv::Mat XZInv = cv::Mat::eye(4, 4, CV_64F);
    XZInv.at<double>(0, 0) = XZInv.at<double>(2, 2) = -1;
    out = XZInv * transform;
    return out;
}

static cv::Mat rot2euler_unity(cv::Mat rot)
{
    // In Unity rotations are performed around the Z axis, the X axis, and the Y axis, in that order.
    double thetaX = 0.0;
    double thetaY = 0.0;
    double thetaZ = 0.0;
    cv::Mat euler = cv::Mat(3, 1, CV_64F);

    double r00 = rot.at<double>(0, 0);
    double r01 = rot.at<double>(0, 1);
    double r02 = rot.at<double>(0, 2);
    double r10 = rot.at<double>(1, 0);
    double r11 = rot.at<double>(1, 1);
    double r12 = rot.at<double>(1, 2);
    double r20 = rot.at<double>(2, 0);
    double r21 = rot.at<double>(2, 1);
    double r22 = rot.at<double>(2, 2);

    // Ry * Rx * Rz * transform order - note that Rz is the first rotation to left of the transform.
    if (r12 < 1)
    {
        if (r12 > -1)
        {
            thetaX = asin(-r12);
            thetaY = atan2(r02, r22);
            thetaZ = atan2(r10, r11);
        }
        else  // r12 = -1
        {
            // not a unique solution: thetaZ - thetaY = atan2(-r01, r00)
            thetaX = M_PI / 2;
            thetaY = -atan2(-r01, r00);
            thetaZ = 0;
        }
    }
    else // r12 = 1
    {
        // Not a unique solution : thetaZ + thetaY = atan2 (-r01, r00)
        thetaX = -M_PI / 2;
        thetaY = atan2(-r01, r00);
        thetaZ = 0;
    }

    // return as degrees
    euler.at<double>(0, 0) = thetaX * (180 / M_PI);
    euler.at<double>(1, 0) = thetaY * (180 / M_PI);
    euler.at<double>(2, 0) = thetaZ * (180 / M_PI);

    return euler;
}


bool CalibrationData::save_calibration_json(QString const& path, int cam_flags, int proj_flags, int stereo_flags, QSettings* config)
{
    // get current timestamp to save to file
    time_t now;
    time(&now);
    char timestamp_str[sizeof "YYYY-MM-DDTHH:MM:SSZ"];
    strftime(timestamp_str, sizeof timestamp_str, "%FT%TZ", gmtime(&now));

    // save capture camera calibration
    QString filename = path + "/capture_camera_calibration.json";
    FILE* fp = fopen(qPrintable(filename), "w");
    if (!fp)
    {
        return false;
    }
    fprintf(fp,
        "{\n"
            "\t\"name\": \"%s\",\n"
            "\t\"timestamp\": \"%s\",\n"
            "\t\"width\": %lf,\n"
            "\t\"height\": %lf,\n"
            "\t\"fx\": %lf,\n"
            "\t\"fy\": %lf,\n"
            "\t\"cx\": %lf,\n"
            "\t\"cy\": %lf,\n"
            "\t\"k1\": %lf,\n"
            "\t\"k2\": %lf,\n"
            "\t\"p1\": %lf,\n"
            "\t\"p2\": %lf,\n"
            "\t\"k3\": %lf,\n"
            "\t\"rx\": %lf,\n"
            "\t\"ry\": %lf,\n"
            "\t\"rz\": %lf,\n"
            "\t\"tx\": %lf,\n"
            "\t\"ty\": %lf,\n"
            "\t\"tz\": %lf\n"
        "}\n",
        filename.toStdString().c_str(),
        timestamp_str,
        cam_width,
        cam_height,
        cam_K.at<double>(0, 0), 
        cam_K.at<double>(1, 1),
        cam_K.at<double>(0, 2),
        cam_K.at<double>(1, 2),
        cam_kc.at<double>(0, 0),
        cam_kc.at<double>(0, 1),
        cam_kc.at<double>(0, 2),
        cam_kc.at<double>(0, 3),
        cam_kc.at<double>(0, 4),
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0
    );
    fclose(fp);

    // process r and t components for unity.
    // create extrinsic matrix
    cv::Mat extrinsics = cv::Mat::eye(4, 4, CV_64F);
    // assign rotation
    R.copyTo(extrinsics(cv::Rect_<int>(0, 0, 3, 3)));
    // assign translation
    extrinsics.at<double>(0, 3) = T.at<double>(0, 0) / 1000.0f;
    extrinsics.at<double>(1, 3) = T.at<double>(1, 0) / 1000.0f;
    extrinsics.at<double>(2, 3) = T.at<double>(2, 0) / 1000.0f;
    std::cout << "Extrinsics:" << std::endl;
    std::cout << extrinsics << std::endl;
    std::cout << std::endl;

    // convert from extrinsic to projector pose
    cv::Mat raw_pose = extrinsics.inv();
    std::cout << "Raw Pose:" << std::endl;
    std::cout << raw_pose << std::endl;
    std::cout << std::endl;

    // convert pose from opencv to unity
    cv::Mat unity_pose = convert_transform_from_opencv_to_unity(raw_pose);
    std::cout << "Unity Pose:" << std::endl;
    std::cout << unity_pose << std::endl;
    std::cout << std::endl;

    // extract rotation
    cv::Mat rot = rot2euler_unity(unity_pose(cv::Rect_<int>(0, 0, 3, 3)));
    std::cout << "Rotation" << std::endl;
    std::cout << rot << std::endl;
    std::cout << std::endl;

    // save projector calibration
    filename = path + "/projector_calibration.json";
    fp = fopen(qPrintable(filename), "w");
    if (!fp)
    {
        return false;
    }
    fprintf(fp,
        "{\n"
        "\t\"name\": \"%s\",\n"
        "\t\"timestamp\": \"%s\",\n"
        "\t\"width\": %lf,\n"
        "\t\"height\": %lf,\n"
        "\t\"fx\": %lf,\n"
        "\t\"fy\": %lf,\n"
        "\t\"cx\": %lf,\n"
        "\t\"cy\": %lf,\n"
        "\t\"k1\": %lf,\n"
        "\t\"k2\": %lf,\n"
        "\t\"p1\": %lf,\n"
        "\t\"p2\": %lf,\n"
        "\t\"k3\": %lf,\n"
        "\t\"rx\": %lf,\n"
        "\t\"ry\": %lf,\n"
        "\t\"rz\": %lf,\n"
        "\t\"tx\": %lf,\n"
        "\t\"ty\": %lf,\n"
        "\t\"tz\": %lf\n"
        "}\n",
        filename.toStdString().c_str(),
        timestamp_str,
        proj_width,
        proj_height,
        proj_K.at<double>(0, 0),
        proj_K.at<double>(1, 1),
        proj_K.at<double>(0, 2),
        proj_K.at<double>(1, 2),
        proj_kc.at<double>(0, 0),
        proj_kc.at<double>(0, 1),
        proj_kc.at<double>(0, 2),
        proj_kc.at<double>(0, 3),
        proj_kc.at<double>(0, 4),
        rot.at<double>(0, 0),
        rot.at<double>(1, 0),
        rot.at<double>(2, 0),
        unity_pose.at<double>(0, 3),
        unity_pose.at<double>(1, 3),
        unity_pose.at<double>(2, 3)
    );
    fclose(fp);

    // format per-view errors for json printing
    std::string cam_pve_str = "";
    cam_pve_str << cam_per_view_errors;
    cam_pve_str = std::regex_replace(cam_pve_str, std::regex("\\n"), " ");

    std::string proj_pve_str = "";
    proj_pve_str << proj_per_view_errors;
    proj_pve_str = std::regex_replace(proj_pve_str, std::regex("\\n"), " ");

    std::string stereo_pve_str = "";
    stereo_pve_str << stereo_per_view_errors;
    stereo_pve_str = std::regex_replace(stereo_pve_str, std::regex("\\n"), " ");

    // save projector calibration report
    filename = path + "/calibration_report.json";
    fp = fopen(qPrintable(filename), "w");
    if (!fp)
    {
        return false;
    }
    fprintf(fp,
        "{\n"
        "\t\"name\": \"%s\",\n"
        "\t\"timestamp\": \"%s\",\n"
        "\t\"capture_camera_rmse\": %lf,\n"
        "\t\"projector_rmse\": %lf,\n"
        "\t\"stereo_rmse\": %lf,\n"
        "\t\"capture_camera_calibration_flags\": %d,\n"
        "\t\"projector_calibration_flags\": %d,\n"
        "\t\"stereo_calibration_flags\": %d,\n"
        "\t\"checkerboard_corner_count_x\": %d,\n"
        "\t\"checkerboard_corner_count_y\": %d,\n"
        "\t\"checkerboard_corners_width\": %lf,\n"
        "\t\"checkerboard_corners_height\": %lf,\n"
        "\t\"decode_threshold\": %d,\n"
        "\t\"decode_b\": %lf,\n"
        "\t\"decode_m\": %d,\n"
        "\t\"calibration_h_win\": %d,\n"
        "\t\"intrinsics_source\": \"%s\",\n"
        "\t\"capture_camera_per_view_error\": \"%s\",\n"
        "\t\"projector_per_view_error\": \"%s\",\n"
        "\t\"stereo_per_view_error\": \"%s\"\n"
        "}\n",
        filename.toStdString().c_str(),
        timestamp_str,
        cam_error,
        proj_error,
        stereo_error,
        cam_flags,
        proj_flags,
        stereo_flags,
        config->value("main/corner_count_x").toUInt(),
        config->value("main/corner_count_y").toUInt(),
        config->value("main/corners_width").toDouble(),
        config->value("main/corners_height").toDouble(),
        config->value(THRESHOLD_CONFIG, THRESHOLD_DEFAULT).toInt(),
        config->value(ROBUST_B_CONFIG, ROBUST_B_DEFAULT).toFloat(),
        config->value(ROBUST_M_CONFIG, ROBUST_M_DEFAULT).toUInt(),
        config->value(HOMOGRAPHY_WINDOW_CONFIG, HOMOGRAPHY_WINDOW_DEFAULT).toUInt(),
        config->value(INTRINSICS_SOURCE_CONFIG, QString(INTRINSICS_SOURCE_DEFAULT)).toString().toStdString().c_str(),
        cam_pve_str.c_str(),
        proj_pve_str.c_str(),
        stereo_pve_str.c_str()
        );
    fclose(fp);
    return true;
}

bool CalibrationData::save_calibration_matlab(QString const& filename)
{
    FILE * fp = fopen(qPrintable(filename), "w");
    if (!fp)
    {
        return false;
    }

    cv::Mat rvec;
    cv::Rodrigues(R, rvec);
    fprintf(fp, 
        "%% Projector-Camera Stereo calibration parameters:\n"
        "\n"
        "%% Intrinsic parameters of camera:\n"
        "fc_left = [ %lf %lf ]; %% Focal Length\n"
        "cc_left = [ %lf %lf ]; %% Principal point\n"
        "alpha_c_left = [ %lf ]; %% Skew\n"
        "kc_left = [ %lf %lf %lf %lf %lf ]; %% Distortion\n"
        "\n"
        "%% Intrinsic parameters of projector:\n"
        "fc_right = [ %lf %lf ]; %% Focal Length\n"
        "cc_right = [ %lf %lf ]; %% Principal point\n"
        "alpha_c_right = [ %lf ]; %% Skew\n"
        "kc_right = [ %lf %lf %lf %lf %lf ]; %% Distortion\n"
        "\n"
        "%% Extrinsic parameters (position of projector wrt camera):\n"
        "om = [ %lf %lf %lf ]; %% Rotation vector\n"
        "T = [ %lf %lf %lf ]; %% Translation vector\n",
        cam_K.at<double>(0,0), cam_K.at<double>(1,1), cam_K.at<double>(0,2), cam_K.at<double>(1,2), cam_K.at<double>(0,1),
        cam_kc.at<double>(0,0), cam_kc.at<double>(0,1), cam_kc.at<double>(0,2), cam_kc.at<double>(0,3), cam_kc.at<double>(0,4), 
        proj_K.at<double>(0,0), proj_K.at<double>(1,1), proj_K.at<double>(0,2), proj_K.at<double>(1,2), proj_K.at<double>(0,1),
        proj_kc.at<double>(0,0), proj_kc.at<double>(0,1), proj_kc.at<double>(0,2), proj_kc.at<double>(0,3), proj_kc.at<double>(0,4),
        rvec.at<double>(0,0), rvec.at<double>(1,0), rvec.at<double>(2,0), 
        T.at<double>(0,0), T.at<double>(1,0), T.at<double>(2,0)
        );
    fclose(fp);

    return true;
}

void CalibrationData::display(std::ostream & stream) const
{
    stream << "Camera Calib: " << std::endl
        << " - reprojection error: " << cam_error << std::endl
        << " - K:\n" << cam_K << std::endl
        << " - kc: " << cam_kc << std::endl
        ;

    stream << std::endl;
    stream << "Projector Calib: " << std::endl
        << " - reprojection error: " << proj_error << std::endl
        << " - K:\n" << proj_K << std::endl
        << " - kc: " << proj_kc << std::endl
        ;

    // Log the per-view errors (per-view error dim: n x 1)
    stream << " - per-view reprojection errors: [" << "\n";
    for (int r = 0; r < proj_per_view_errors.rows; ++r)
        stream << r << ": " << proj_per_view_errors.row(r) << '\n';
    stream << "]" << std::endl;

    stream << std::endl;
    stream << "Stereo Calib: " << std::endl
        << " - reprojection error: " << stereo_error << std::endl
        << " - R:\n" << R << std::endl
        << " - T:\n" << T << std::endl
        ;

    // Log the per-view errors (per-view error dim: n x 2)
    stream << " - per-view reprojection errors: [" << "\n";
    for (int r = 0; r < stereo_per_view_errors.rows; ++r)
        stream << r << ": " << stereo_per_view_errors.row(r) << '\n';
    stream << "]" << std::endl;
}

