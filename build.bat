pushd shaders
glslc.exe -O matmul_v1.comp -o matmul_v1.spv
glslc.exe -O matmul_v2.comp -o matmul_v2.spv
glslc.exe -O matmul_v3.comp -o matmul_v3.spv
popd

@REM Build my file
clang .\src\compute.c -std=c11 -o compute.exe -g -Og -IC:\clibs\glfw-3.3.4\include -IC:\VulkanSDK\1.2.182.0\Include C:\clibs\glfw-3.3.4\lib-mingw-w64\libglfw3.a -lopengl32 -luser32 -lgdi32 C:\VulkanSDK\1.2.182.0\Lib\vulkan-1.lib -Wno-writable-strings