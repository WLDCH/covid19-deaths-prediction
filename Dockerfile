
#Base Image to use
FROM python:3.8-slim

#Expose port 8080
EXPOSE 8080
EXPOSE 5000
EXPOSE 8501
EXPOSE 4200
EXPOSE 9515

RUN  apt-get update \
  && apt-get install -y wget \
  && rm -rf /var/lib/apt/lists/*

# install chrome
RUN apt-get update && apt-get install -y gnupg
RUN wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add -
RUN sh -c 'echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google-chrome.list'
RUN apt-get -y update
RUN apt-get install -y google-chrome-stable
RUN google-chrome --version

# install chromedriver
RUN apt-get install -yqq unzip
RUN wget https://chromedriver.storage.googleapis.com/105.0.5195.19/chromedriver_linux64.zip
RUN unzip chromedriver_linux64.zip 
RUN mv chromedriver /usr/bin/chromedriver 
RUN chmod +x /usr/bin/chromedriver 

# set display port to avoid crash
ENV DISPLAY=:99

COPY . .

RUN pip3 install --upgrade pip
RUN pip install pipenv
RUN pipenv install

#Change Working Directory to app directory
WORKDIR .

#Run the application on port 8080
ENTRYPOINT ["pipenv", "run", "streamlit", "run", "dashboard/app.py", "--server.port=8080", "--server.address=0.0.0.0"]
