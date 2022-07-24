OpenVINO
=====================

OpenVINO™ is an open-source toolkit for optimizing and deploying AI inference.


System requirements
-------------------

- Python >= 3.6


Installation
------------

Criar pastas com o nome: lib, model
## Copiar na pasta lib os arquivos:
	- libcpu_extension.so
	- libformat_reader.so
	- libgflags_nothreads.a
### Criar dentra da pasta model as pastas:
	- age-gender
		- FP16
			- age-gender-recognition.bin # Copiar arquivo
			- age-gender-recognition.xml # Copiar arquivo
		- FP32
			- age-gender-recognition.bin # Copiar arquivo
			- age-gender-recognition.bin # Copiar arquivo
	- emotions
		- FP16
			- emotions-recognition.bin # Copiar arquivo
			- emotions-recognition.xml # Copiar arquivo
		- FP32
			- emotions-recognition.bin # Copiar arquivo
			- emotions-recognition.xml # Copiar arquivo
		- INT8
			- emotions-recognition.bin # Copiar arquivo
			- emotions-recognition.xml # Copiar arquivo
	- face-detection
		- FP16
			- face-detection.bin # Copiar arquivo
			- face-detection.xml # Copiar arquivo
		- FP32
			- face-detection.bin # Copiar arquivo
			- face-detection.xml # Copiar arquivo
		- INT8
			- face-detection.bin # Copiar arquivo
			- face-detection.xml # Copiar arquivo
	- facial-landmarks
		- FP16
			- facial-landmarks.bin # Copiar arquivo
			- facial-landmarks.xml # Copiar arquivo
		- FP32
			- facial-landmarks.bin # Copiar arquivo
			- facial-landmarks.xml # Copiar arquivo
	- head-pose
		- FP16
			- head-pose-estimation.bin # Copiar arquivo
			- head-pose-estimation.xml # Copiar arquivo
		- FP32
			- head-pose-estimation.bin # Copiar arquivo
			- head-pose-estimation.xml # Copiar arquivo

## Procurar os modelos na instalação de openvino. Os nomes poden no ser do mesmo jeito.

```bash
$ sudo mkdir /var/log/openvino  # creates openvino logging folder
$ sudo chown $USER:$USER /var/log/openvino  # give openvino logging folder group permissions
$ pip install -e .
```


Usage
-----

```bash
$ ov  # run API
```

Read REST API documentation through ``/docs`` endpoint for API usage.
