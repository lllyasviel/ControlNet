pipeline {
    agent any

    environment {
        PATH = "${PATH}:${HOME}/.local/bin"
    }

    stages {
        stage('Checkout SCM') {
            steps {
                checkout([$class: 'GitSCM',
                    branches: [[name: '*/i192011']],
                    userRemoteConfigs: [[url: 'https://github.com/hmzakhalid/ControlNet.git']]])
            }
        }

        stage('Lint with flake8') {
            steps {
                sh 'pip3 install flake8'
                sh 'flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics'
            }
        }

        stage('Format with black') {
            steps {
                sh 'pip3 install black'
                sh 'black .'
            }
        }
    }
}