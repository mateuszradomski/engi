@REM C:\VulkanSDK\1.2.182.0\Bin32\glslc.exe -fshader-stage=vertex vertex.glsl -o vertex.spv
@REM C:\VulkanSDK\1.2.182.0\Bin32\glslc.exe -fshader-stage=fragment pixel.glsl -o pixel.spv

@REM Build base file
zig c++ .\src\lodepng.cpp .\src\main.cpp -o mandelbrot.exe -g -Og -IC:\clibs\glfw-3.3.4\include -IC:\VulkanSDK\1.2.182.0\Include C:\clibs\glfw-3.3.4\lib-mingw-w64\libglfw3.a -lopengl32 -luser32 -lgdi32 C:\VulkanSDK\1.2.182.0\Lib\vulkan-1.lib -Wno-writable-strings

@REM Build my file
zig cc .\src\compute.c -std=c11 -o compute.exe -g -Og -IC:\clibs\glfw-3.3.4\include -IC:\VulkanSDK\1.2.182.0\Include C:\clibs\glfw-3.3.4\lib-mingw-w64\libglfw3.a -lopengl32 -luser32 -lgdi32 C:\VulkanSDK\1.2.182.0\Lib\vulkan-1.lib -Wno-writable-strings

pushd shaders
glslangValidator.exe -V shader.comp
popd