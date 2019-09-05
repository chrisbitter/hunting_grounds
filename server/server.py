import paramiko

ssh_client = paramiko.SSHClient()
ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh_client.connect(hostname="137.226.188.42", username="cscheiderer",
                   password="Y#Imaany4$ZlwK")

res = ssh_client.exec_command("cd hunting grounds && git pull")

# todo[?]: in den server folder navigieren und git clone, git pull, git checkout -b server machen
# todo[?]: funktion, die alle relevanten Daten vom Server synchronisiert
# todo[?]: alles in eine Klasse packen
# todo[?]: aufräumen. dokumentieren. nochmal aufräumen

if __name__ == "__main__":
    ...
