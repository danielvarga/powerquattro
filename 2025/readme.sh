
mkdir Data250307.pkls
ls Data250307/*.bin | while read f ; do out=$(echo $f | sed "s/\//.pkls\//" | sed "s/bin$/pkl/") ; python binary_reader.py $f $out ; done
