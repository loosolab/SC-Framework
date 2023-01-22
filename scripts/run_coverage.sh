dir=$(pwd)
echo $dir

mkdir ~/coverage_run
cp -r tests ~/coverage_run/.
cp -r sctoolbox ~/coverage_run/. 

cd ~/coverage_run  # pytest does not work on agnerds

pytest --cov-report html --cov=sctoolbox --ignore=tests/data/* tests/*
cp -r htmlcov "${dir}/." #copy htmlcov to the original directory

rm -r ~/coverage_run