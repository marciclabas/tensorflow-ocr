from typing_extensions import TypedDict, NotRequired, Sequence, overload
import aiohttp
from haskellian import Either, Left, Right
from .types import TFSResponse, ImagePreds

class Params(TypedDict):
  host: NotRequired[str]
  port: NotRequired[int]
  endpoint: NotRequired[str]

PredictErr = aiohttp.ClientError

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
        return Right(TFSResponse.model_validate_json(x).predictions)
      
  except aiohttp.ClientError as e:
    return Left(e)
    