
new_path="/mounts/work/ayyoob/alignment/output/"
old_path="/mounts/work/ayyoob/alignment/output/1/"

files=`ls $old_path`

for file in $files 
do
    echo "$file"
    mv "$new_path/$file" "$new_path/$file.back"
    mv "$old_path/$file" "$new_path/$file"
done



