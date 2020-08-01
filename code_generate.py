import click
from distutils.dir_util import copy_tree
import os

@click.command()
@click.option('--code_path', default="./test",
              help='path to generate training code')
def main(code_path):
    template_path = "./templates"
    copy_tree(template_path, code_path)
    os.mkdir(os.path.join(code_path, 'output'))



if __name__ == '__main__':
    main()
