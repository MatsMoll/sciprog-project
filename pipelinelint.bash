pylint kode/*.py || true # Ignorerer exit code
values=$(pylint kode/*.py | grep rated | awk '{print $7}')
IFS='/'
score=$(echo $values | awk '{print $1}')
if (( $(echo "$score > 5" | wc -l) )); then
    echo "Code rating of $score/10"
    exit 0
else
    echo "To low code rating $score/10"
    exit 1
fi