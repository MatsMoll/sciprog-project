pylint kode/*.py --rcfile=lint-config.rc || true # Ignoring exit code
values=$(pylint kode/*.py --rcfile=lint-config.rc | grep rated | awk '{print $7}')
IFS='/'
score=$(echo $values | awk '{print $1}')
IFS='.'
score=$(echo $score | awk '{print $1$2}')
if [ "$score" -lt "950" ]; then
    echo "To low code rating $score < 950"
    exit 1
fi

cd kode
coverage run align_image_test.py
coverage run hdr_test.py
coverage run globalHDR_test.py
coverage run localHDR_test.py
coverage report
pros=$(coverage report | grep TOTAL | awk '{print $4}')
pros=${pros%\%}
echo $pros
if [ "$pros" -lt "50" ]; then
    echo "To low coverage $pros < 50"
    exit 1
fi
exit 0
