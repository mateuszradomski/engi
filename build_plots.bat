@REM @REM Generate matrix plots
@REM pushd build
@REM python plot.py
@REM popd

@REM Generate barcharts
pushd plots
python barplot_gflop.py
python barplot_memorytime.py
popd
