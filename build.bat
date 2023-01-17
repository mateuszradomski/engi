if not exist "build" mkdir build

@REM Build my file
pushd build

if not exist "shaders" mkdir shaders

pushd shaders
glslc.exe -O -fshader-stage=comp ..\..\shaders\sparse_matmul_ell.glsl -o sparse_matmul_ell.spv
glslc.exe -O -fshader-stage=comp ..\..\shaders\sparse_matmul_ell_offset.glsl -o sparse_matmul_ell_offset.spv
glslc.exe -O -fshader-stage=comp ..\..\shaders\sparse_matmul_sell.glsl -o sparse_matmul_sell.spv
glslc.exe -O -fshader-stage=comp ..\..\shaders\sparse_matmul_coo.glsl -o sparse_matmul_coo.spv
glslc.exe -O -fshader-stage=comp ..\..\shaders\sparse_matmul_csr.glsl -o sparse_matmul_csr.spv
glslc.exe -O -fshader-stage=comp ..\..\shaders\sparse_matmul_csc.glsl -o sparse_matmul_csc.spv
glslc.exe -O -fshader-stage=comp ..\..\shaders\sparse_matmul_bsr.glsl -o sparse_matmul_bsr.spv
popd

set "COMP_FLAGS=-fsanitize=address -fsanitize=undefined"
@REM set "COMP_FLAGS="

clang ..\src\compute.c -std=c11 -o compute.exe ^
-g %COMP_FLAGS% ^
-IC:\clibs\glfw-3.3.4\include ^
-IC:\VulkanSDK\1.2.182.0\Include ^
C:\clibs\glfw-3.3.4\lib-mingw-w64\libglfw3.a ^
-lopengl32 -luser32 -lgdi32 C:\VulkanSDK\1.2.182.0\Lib\vulkan-1.lib -Wno-writable-strings
popd
