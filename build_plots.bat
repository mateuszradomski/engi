@REM @REM Generate matrix plots
@REM pushd build
@REM python plot.py
@REM popd

@REM Generate barcharts
python plots\barplot_gflop.py
python plots\barplot_memorytime.py
python plots\barplot_memorytput.py
python plots\barplot_memory_effective_tput.py
