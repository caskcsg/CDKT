curl -X 'POST' \
  'http://192.168.10.75:5000/model/predict' \
  -H 'accept: /workspace/hateful_memes/img' \
  -H 'Content-Type: multipart/form-data' \
  -F 'image=@02165.png;type=image/png'
