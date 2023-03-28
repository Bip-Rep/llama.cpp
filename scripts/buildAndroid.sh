#!/bin/bash
#add NDK path to your .bash_profile
echo $NDK
rm -r buildAndroid
mkdir buildAndroid
cd buildAndroid
cmake -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-23 -DCMAKE_C_FLAGS=-march=armv8.4a+dotprod -DBUILD_SHARED_LIBS=true ..
make
# mv *.so ../../../assets/libs/