## Initial setup process

1. Clone the GitHub repository
```
git clone git@github.com:OpenSenseLabs/nlp-tools.git
```
2. Move into the ```nlp-tools``` directory
```
cd nlp-tools
```
3. Update the list of packages
```
sudo apt update
```
4. Install ```python3``` and ```pip3```
```
sudo apt install python3-minimal python3-pip
```
5. Install all the required Python packages mentioned within ```requirements.txt```
```
pip3 install -r requirements.txt
```
6. Download all the required NLTK datasets/models (```averaged_perceptron_tagger```, ```punkt```, and ```vader_lexicon```)
```
python3 nltk.py
```
7. Run the Python script (Server runs on ```http://0.0.0.0:5000/``` by default)
```
python3 app.py
```
8. Perform HTTP POST requests using cURL in a new terminal tab/window or use a client like Postman to make sure everthing works correctly

```
curl -X POST \
  http://0.0.0.0:5000/summarize \
  -H 'Postman-Token: 6d3a96a8-1bdb-4dd3-ba7c-7ce7902f5152' \
  -H 'cache-control: no-cache' \
  -H 'content-type: multipart/form-data; boundary=----WebKitFormBoundary7MA4YWxkTrZu0gW' \
  -F 'text=My long text here. Make sure to escape the special characters'
```
```
curl -X POST \
  http://0.0.0.0:5000/keyword \
  -H 'Postman-Token: 6d3a96a8-1bdb-4dd3-ba7c-7ce7902f5152' \
  -H 'cache-control: no-cache' \
  -H 'content-type: multipart/form-data; boundary=----WebKitFormBoundary7MA4YWxkTrZu0gW' \
  -F 'text=My long text here. Make sure to escape the special characters'
```
```
curl -X POST \
  http://0.0.0.0:5000/duplicate \
  -H 'Postman-Token: 6d3a96a8-1bdb-4dd3-ba7c-7ce7902f5152' \
  -H 'cache-control: no-cache' \
  -H 'content-type: multipart/form-data; boundary=----WebKitFormBoundary7MA4YWxkTrZu0gW' \
  -F 'text=My long text here. Make sure to escape the special characters'
```
