#This file is for any packages you need installed for your skill to run
if [ ! -d "/opt/mycroft/habits" ]; then
    sudo mkdir /opt/mycroft/habits
    touch /opt/mycroft/habits/logs.json
    echo '[]' > /opt/mycroft/habits/habits.json
    echo '[]' > /opt/mycroft/habits/triggers.json
    sudo chmod -R ugo+rw /opt/mycroft/habits
fi

sudo apt-get install python-pip
#pip install --upgrade pip
#pip install -Iv scipy==0.13.3
#pip install -U scikit-learn
#pip install -U statistics
#pip install -U numpy
