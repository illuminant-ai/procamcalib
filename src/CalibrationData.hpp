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

#ifndef __CALIBRATIONDATA_HPP__
#define __CALIBRATIONDATA_HPP__

#include <iostream>
#include <QString>
#include <QSettings>
#include <vector>
#include <opencv2/core.hpp>

class CalibrationData
{
public:
    static const int CALIBRATION_FILE_VERSION = 1;

    CalibrationData();
    ~CalibrationData();

    void clear(void);

    bool is_valid(void) const;

    bool load_calibration(QString const& filename);
    bool save_calibration(QString const& filename);

    bool load_calibration_yml(QString const& filename);
    bool save_calibration_yml(QString const& filename);

    bool save_calibration_matlab(QString const& filename);
    bool save_calibration_json(QString const& path, int cam_flags, int proj_flags, int stereo_flags, QSettings* config);

    void display(std::ostream & stream = std::cout) const;

    //data
    cv::Mat cam_K;
    cv::Mat cam_kc;
    cv::Mat proj_K;
    cv::Mat proj_kc;
    cv::Mat R;
    cv::Mat T;
    cv::Mat E;
    cv::Mat F;
    // cv::Mat H1;
    // cv::Mat H2;

    double cam_error;
    double proj_error;
    double stereo_error;

    double cam_width;
    double cam_height;
    double proj_width;
    double proj_height;

    cv::Mat cam_per_view_errors;
    cv::Mat proj_per_view_errors;
    cv::Mat stereo_per_view_errors;

    QString filename;
};

#endif //__CALIBRATIONDATA_HPP__