for file in src/*.png; do
convert -scale 7000% -bordercolor black -border 1% $file "`basename $file .png`.png";
done