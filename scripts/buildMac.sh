#!/bin/bash
rm -r buildMac
mkdir buildMac
cd buildMac
cmake -DBUILD_SHARED_LIBS=true ..
make
mv libllama.dylib ../../../packages/llama/macos/Frameworks