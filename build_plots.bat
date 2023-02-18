@REM @REM Generate matrix plots
@REM pushd build
@REM python plot.py
@REM popd

@REM Generate barcharts
python barplot_gflop.py
python barplot_memorytime.py
python barplot_memorytput.py
python barplot_memory_effective_tput.py
