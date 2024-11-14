line=https://www.youtube.com/watch?v=-_e5KvSYXLA
video_id=$(echo "$line" | grep -oP '(?<=watch\?v=)[^&]+' | tr -d '\n' | tr -d '\r')
echo "video_id: $video_id"
echo "Checking if file exists: /mnt/wenwan/data/speech/raw/yt/241111/$video_id.opus"
new_filepath=temp.txt
while IFS= read -r line
  do
    echo "Line $line_number: $line"
    video_id=$(echo "$line" | grep -oP '(?<=watch\?v=)[^&]+' | tr -d '\n' )
    echo "video_id: $video_id"
    echo "Checking if file exists: /mnt/wenwan/data/speech/raw/yt/241111/$video_id.opus"
  done < $new_filepath
done
