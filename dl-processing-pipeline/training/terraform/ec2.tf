resource "aws_instance" "web" {
  ami                    = "ami-0866a3c8686eaeeba" # ubuntu
  instance_type          = "t3.micro"
  key_name               = "terraform-ec2-key"
  vpc_security_group_ids = [aws_security_group.security_group_terraform.id]

  tags = {
    Name = "terraform-app-ec2"
  }
    user_data = <<-EOF
              #!/bin/bash
              # Update packages
              sudo apt-get update -y

              # Install git
              sudo apt-get install git -y

              mkdir -p /workspace
              cd /workspace

              # Clone the GitHub repository
              git clone https://github.com/rafael-curran/CS_598_CCC

              # Navigate into the repository
              cd CS_598_CCC

              # Run a command, e.g., start a script
              EOF
}