if not exist "build" mkdir build

@REM Build my file
pushd build

if not exist "shaders" mkdir shaders

pushd shaders
@REM glslc.exe -O ..\..\shaders\matmul_v1.comp -o matmul_v1.spv
@REM glslc.exe -O ..\..\shaders\matmul_v2.comp -o matmul_v2.spv
@REM glslc.exe -O ..\..\shaders\matmul_v3.comp -o matmul_v3.spv
glslc.exe -O -fshader-stage=comp ..\..\shaders\sparse_matmul_v1.glsl -o sparse_matmul_v1.spv
glslc.exe -O -fshader-stage=comp ..\..\shaders\sparse_matmul_v2.glsl -o sparse_matmul_v2.spv
glslc.exe -O -fshader-stage=comp ..\..\shaders\sparse_matmul_v3.glsl -o sparse_matmul_v3.spv
glslc.exe -O -fshader-stage=comp ..\..\shaders\sparse_matmul_coo.glsl -o sparse_matmul_coo.spv
glslc.exe -O -fshader-stage=comp ..\..\shaders\sparse_matmul_csr.glsl -o sparse_matmul_csr.spv
glslc.exe -O -fshader-stage=comp ..\..\shaders\sparse_matmul_csc.glsl -o sparse_matmul_csc.spv
glslc.exe -O -fshader-stage=comp ..\..\shaders\sparse_matmul_bsr.glsl -o sparse_matmul_bsr.spv
popd

@REM -fsanitize=address ^
@REM -fsanitize=undefined ^

clang ..\src\compute.c -std=c11 -o compute.exe ^
-g ^
-IC:\clibs\glfw-3.3.4\include ^
-IC:\VulkanSDK\1.2.182.0\Include ^
C:\clibs\glfw-3.3.4\lib-mingw-w64\libglfw3.a ^
-lopengl32 -luser32 -lgdi32 C:\VulkanSDK\1.2.182.0\Lib\vulkan-1.lib -Wno-writable-strings
popd
