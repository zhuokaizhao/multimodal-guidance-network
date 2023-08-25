#!/bin/bash

# /home/ec2-user/anaconda3/envs/JupyterSystemEnv/bin/pip install -U keytar jupyter-server-proxy
echo == INSTALLING CODE-SERVER ==
curl -fsSL https://code-server.dev/install.sh | sh -s -- --version=4.12.0

#########################################
### INTEGRATE CODE-SERVER WITH JUPYTER
#########################################
echo == UPDATING THE JUPYTER SERVER CONFIG ==
cat >>/home/ec2-user/.jupyter/jupyter_notebook_config.py <<EOC
c.ServerProxy.servers = {
  'vscode': {
      'launcher_entry': {
            'enabled': True,
            'title': 'VS Code',
      },
      'command': ['code-server', '--auth', 'none', '--disable-telemetry', '--bind-addr', '0.0.0.0:{port}'],
      'environment' : {'XDG_DATA_HOME' : '/home/ec2-user/SageMaker/vscode-config'},
      'absolute_url': False,
      'timeout': 30
  }
}
EOC


echo == INSTALL SUCCESSFUL. RESTARTING JUPYTER ==
# RESTART THE JUPYTER SERVER
systemctl restart jupyter-server