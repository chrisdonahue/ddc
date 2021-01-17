docker run \
  -it \
  -p 8080:80 \
	-v songs:/ddc/songs \
  chrisdonahue/ddc:latest \
  python ddc_server.py
