#!/bin/bash
rm -r buildIos
mkdir buildIos
cd buildIos
cmake -G Xcode -B build -DCMAKE_SYSTEM_NAME=iOS -DIOS_INSTALL_COMBINED=true -DCMAKE_Swift_COMPILER_FORCED=true -DCMAKE_OSX_DEPLOYMENT_TARGET=11.0 -DBUILD_SHARED_LIBS=true ..
cd build
xcodebuild -scheme ALL_BUILD -sdk iphoneos
# mv Debug-iphoneos/libllama.dylib ../../../../packages/llama/ios/Frameworks/