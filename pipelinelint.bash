pylint kode/*.py --rcfile=lint-config.rc || true # Ignorerer exit code
values=$(pylint kode/*.py --rcfile=lint-config.rc | grep rated | awk '{print $7}')
IFS='/'
score=$(echo $values | awk '{print $1}')
if (( $(echo "$score > 9.5" | bc -l) )); then
    exit 0
else
    echo "To low code rating $score < 9.5"
    exit 1
fi