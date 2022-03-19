pushd shaders
glslangValidator.exe -V shader.comp -o shader.spirv
glslangValidator.exe -V shader_faster.comp -o shader_faster.spirv
popd

@REM Build base file
@REM zig c++ .\src\lodepng.cpp .\src\main.cpp -o mandelbrot.exe -g -Og -IC:\clibs\glfw-3.3.4\include -IC:\VulkanSDK\1.2.182.0\Include C:\clibs\glfw-3.3.4\lib-mingw-w64\libglfw3.a -lopengl32 -luser32 -lgdi32 C:\VulkanSDK\1.2.182.0\Lib\vulkan-1.lib -Wno-writable-strings

@REM Build my file
clang .\src\compute.c -std=c11 -o compute.exe -g -Og -IC:\clibs\glfw-3.3.4\include -IC:\VulkanSDK\1.2.182.0\Include C:\clibs\glfw-3.3.4\lib-mingw-w64\libglfw3.a -lopengl32 -luser32 -lgdi32 C:\VulkanSDK\1.2.182.0\Lib\vulkan-1.lib -Wno-writable-strings