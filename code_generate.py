import click
from distutils.dir_util import copy_tree


@click.command()
@click.option('--code_path', default="./test",
              help='path to generate training code')
def main(code_path):
    template_path = "./templates"
    copy_tree(template_path, code_path)


if __name__ == '__main__':
    main()
