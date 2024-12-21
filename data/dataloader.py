from peewee import *
import base64
from torch.utils.data import Dataset

db = SqliteDatabase('2021-07-31-lichess-evaluations-37MM.db')

class Evaluations(Model):
  id = IntegerField()
  fen = TextField()
  binary = BlobField()
  eval = FloatField()

  class Meta:
    database = db

  def binary_base64(self):
    return base64.b64encode(self.binary)

db.connect()
LABEL_COUNT = 37164639

def print_query_example():
    entry = Evaluations.get(Evaluations.id == LABEL_COUNT)
    print(entry.binary_base64())
    print(entry.fen)
    print(entry.eval)

class LichessDataset(Dataset):
    def __init__(self, mode=None):
       self.mode = mode
       
    def __len__(self):
        if self.mode == 'train':
            return 26000000
        elif self.mode == 'val':
            return 7500000
        elif self.mode == 'test':
            return 3664639
                
    def __getitem__(self, idx):
        offset = 1
        if self.mode == 'val':
            offset += 25999999
        elif self.mode == 'test':
            offset += 25999999 + 7500000
        
        entry = Evaluations.get(Evaluations.id == (idx + offset))
        return entry.fen, entry.eval 
        
            
