pylint *.py --rcfile=lint-config.rc || true # Ignorerer exit code
values=$(pylint *.py --rcfile=lint-config.rc | grep rated | awk '{print $7}')
IFS='/'
score=$(echo $values | awk '{print $1}')
echo $score
IFS='.'
score=$(echo $score | awk '{print $1$2}')
echo $score
if [ "$score" -gt "950" ]; then
    exit 0
else
    echo "To low code rating $((score / 10)) < 9.5"
    exit 1
fi