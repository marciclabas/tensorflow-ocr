from typing_extensions import TypedDict, NotRequired, Sequence, Literal
from dataclasses import dataclass
import aiohttp
from haskellian import Either, Left, Right
from .types import TFSResponse, TFSOk, TFSError, ImagePreds

class Params(TypedDict):
  host: NotRequired[str]
  port: NotRequired[int]
  endpoint: NotRequired[str]

@dataclass
class ConnectionError:
  error: aiohttp.ClientError
  tag: Literal['connection-error'] = 'connection-error'

@dataclass
class TFServingError:
  error: str
  tag: Literal['tfserving-error'] = 'tfserving-error'

PredictErr = ConnectionError | TFServingError

async def predict(
  b64imgs: Sequence[str], *,
  host: str = 'http://localhost',
  port: int = 8501,
  endpoint: str = '/v1/models/ocr:predict'
) -> Either[PredictErr, Sequence[ImagePreds]]:
  """Each `b64imgs[i]` is a base64-encoded JPG/PNG/WEBP image"""
  base = f'{host.strip("/")}:{port}'
  try:
    async with aiohttp.ClientSession(base) as session:
      req = session.post(endpoint, json={
        "signature_name": "serving_default",
        "instances": b64imgs
      })
      async with req as res:
        x = await res.text()
        match TFSResponse.model_validate_json(x).root:
          case TFSOk() as ok:
            return Right(ok.predictions)
          case TFSError() as err:
            return Left(TFServingError(err.error))
      
  except aiohttp.ClientError as e:
    return Left(ConnectionError(e))
    