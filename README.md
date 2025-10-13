# Project Overview 📋

This Python based application provides comprehensive telecommunications network analysis by detecting and monitoring both **GSM** and **UMTS** calls in real time for a very short period of time. The core functionality includes intelligent call detection across multiple network standards, automatic clustering of detected calls based on their respective mobile operators (*Orange*, *IAM*, *Inwi*), and accurate estimation of the total number of active users within a specified range which depend mainely on the antenna type and geometery.

# Usage 🔍

This application is designed to detect and monitor unauthorized phone call activity during examination periods. The system identifies individuals attempting to use mobile devices for cheating purposes by detecting active GSM and UMTS calls within the examination area.

# Hardware Requirement 💻

This code has been tested and optimized for the following hardware configuration:

* Raspberry Pi 5 (Broadcom BCM2712 quad-core Arm Cortex A76 processor).
* USRP B210 Software Defined Radio.

The B210 SDR collects raw data from the cellular network spectrum, which is then processed by the Raspberry Pi 5 to identify and analyze call activity.

