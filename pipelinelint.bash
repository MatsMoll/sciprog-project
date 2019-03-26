pylint kode/*.py --rcfile=lint-config.rc || true # Ignorerer exit code
values=$(pylint kode/*.py --rcfile=lint-config.rc | grep rated | awk '{print $7}')
IFS='/'
score=$(echo $values | awk '{print $1}')
IFS='.'
score=$(echo $score | awk '{print $1$2}')
if [ "$score" -lt "950" ]; then
    echo "To low code rating $score < 950"
    exit 1
fi

coverage run --source=. -m unittest && coverage report | grep TOTAL | awk '{print $3}'
pros=$(coverage run --source=. -m unittest && coverage report | grep TOTAL | awk '{print $3}')
pros=${pros::-1}
if [ "$pros" -lt "70" ]; then
    echo "To low coverage $pros < 70"
    exit 1
fi
exit 0