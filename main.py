


from models import model_1
import click
@click.command()
@click.argument('model', type=click.Choice(['model_1','linear', 'logistic']), default='model_1',)
# @click.option('-n','--number',  type=int, help="please enter the number of time you want to show the message")
def main(model):
    '''
     help message 
    '''

    if model is None:
        model = input('please enter your name: ')
    if model == 'model_1':
        click.echo('model_1 ')
        print(model_1(1,2,3,4))


if __name__ == "__main__":
    main()