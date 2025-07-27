from dataclasses import dataclass
from src.data_models import EvalScriptType
@dataclass
class RequestType():
  evalscript: str
  response: list

rgb_evalscript = """
//VERSION=3

function setup() {
  return {
    input: ["B02", "B03", "B04"],
    output: { id: 'default',
              bands: 3}
  };
}

function evaluatePixel(sample) {
  return [2.5 * sample.B04, 2.5 * sample.B03, 2.5 * sample.B02];
}
"""
rgb_response = [
                {
                    'identifier': 'default',
                    'format': {
                        'type': 'image/jpeg',
                    }
                }
              ]


all_bands_evalscript = """
//VERSION=3

function setup() {
  return {
    input: ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B11", "B12", "SCL"],
    output: [{ id: 'bands',
              bands: 9,
              sampleType: 'FLOAT32'},
              
              { id: 'scl',
              bands: 1,
              sampleType: 'UINT8'}
              ]
  };
}

function evaluatePixel(sample) {
  return { bands: [sample.B02, sample.B03, sample.B04, sample.B05, sample.B06, sample.B07, sample.B08, sample.B11, sample.B12],
           scl: [sample.SCL]
          };
}
"""

all_bands_response = [
                      {
                          'identifier': 'bands',
                          'format': {
                              'type': 'image/tiff',
                          }
                      },
                      {
                          'identifier': 'scl',
                          'format': {
                              'type': 'image/tiff',
                          }
                      }
                     ]


indices_evalscript = """
//VERSION=3

function setup() {
  return {
    input: ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B11", "B12", "SCL"],
    output: [{ id: 'indices',
              bands: 9,
              sampleType: 'FLOAT32'},
              { id: 'scl',
              bands: 1,
              sampleType: 'UINT8'}
              ],
    mosaicking: 'ORBIT'
  };
}

function getFirstQuartile(values) {
  //get the first quartile of the values
    values.sort((a,b) => a-b);
    var index = Math.floor(values.length / 4);
    return values[index];
}

function validate(sample) {
  //define invalid classificated pixels
    const invalid = [
        0, // NO_DATA
        1, // SATURATED_DEFECTIVE
        3, // CLOUD_SHADOW
        7, // CLOUD_LOW_PROBA
        8, // CLOUD_MEDIUM_PROBA
        9, // CLOUD_HIGH_PROBA
        10 // THIN_CIRRUS
    ]
    return !invalid.includes(sample.SCL)
}

function evaluatePixel(samples) {
    var valid = samples.filter(validate);
    if (valid.length > 0 ) {
        let cl = {
            b02: getFirstQuartile(valid.map(s => s.B02)),
            b03: getFirstQuartile(valid.map(s => s.B03)),
            b04: getFirstQuartile(valid.map(s => s.B04)),
            b05: getFirstQuartile(valid.map(s => s.B05)),
            b06: getFirstQuartile(valid.map(s => s.B06)),
            b07: getFirstQuartile(valid.map(s => s.B07)),
            b08: getFirstQuartile(valid.map(s => s.B08)),
            b11: getFirstQuartile(valid.map(s => s.B11)),
            b12: getFirstQuartile(valid.map(s => s.B12)),
        };

        let l = 0.5;

        let saviDenom = cl.b08 + cl.b04 + l;
        let savi = saviDenom !== 0 ? (((cl.b08 - cl.b04) / saviDenom) * (1 + l)) : 0;

        let eviDenom = cl.b08 + 6 * cl.b04 - 7.5 * cl.b02 + 1;
        let evi = eviDenom !== 0 ? (2.5 * (cl.b08 - cl.b04) / eviDenom) : 0;

        let ndre705Denom = cl.b08 + cl.b05;
        let ndre705 = ndre705Denom !== 0 ? ((cl.b08 - cl.b05) / ndre705Denom) : 0;

        let ndre740Denom = cl.b08 + cl.b06;
        let ndre740 = ndre740Denom !== 0 ? ((cl.b08 - cl.b06) / ndre740Denom) : 0;

        let ndre783Denom = cl.b08 + cl.b07;
        let ndre783 = ndre783Denom !== 0 ? ((cl.b08 - cl.b07) / ndre783Denom) : 0;

        let ndviDenom = cl.b08 + cl.b04;
        let ndvi = ndviDenom !== 0 ? ((cl.b08 - cl.b04) / ndviDenom) : 0;

        let ndwigaoDenom = cl.b08 + cl.b11;
        let ndwigao = ndwigaoDenom !== 0 ? ((cl.b08 - cl.b11) / ndwigaoDenom) : 0;

        let ndwimcfDenom = cl.b03 + cl.b08;
        let ndwimcf = ndwimcfDenom !== 0 ? ((cl.b03 - cl.b08) / ndwimcfDenom) : 0;

        let nbrDenom = cl.b08 + cl.b12;
        let nbr = nbrDenom !== 0 ? ((cl.b08 - cl.b12) / nbrDenom) : 0;

        return {
            indices: [savi, evi, ndre705, ndre740, ndre783, ndvi, ndwigao, ndwimcf, nbr],
            scl: [samples[0].SCL]
        };
    }

    // If there isn't enough data, return NODATA
    return {
        indices: [NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN],
        scl: [NaN]
    };
}
"""
indices_response = [
                      {
                          'identifier': 'indices',
                          'format': {
                              'type': 'image/tiff',
                          }
                      },
                      {
                          'identifier': 'scl',
                          'format': {
                              'type': 'image/tiff',
                          }
                      }
                     ]

_requests = {
  "RGB" : RequestType(rgb_evalscript, rgb_response),
  "ALL" : RequestType(all_bands_evalscript, all_bands_response),
  "INDICES": RequestType(indices_evalscript, indices_response)
}

def get_evalscript(mode: EvalScriptType) -> str:
    try:
        return _requests[mode].evalscript
    except KeyError:
        raise KeyError(f"{mode} is not a valid Evalscript Type. Valid types are: {list(_requests.keys())}")

def get_response_setup(mode: EvalScriptType) -> list[dict]:
    try:
        return _requests[mode].response
    except KeyError:
        raise KeyError(f"{mode} is not a valid Evalscript Type. Valid types are: {list(_requests.keys())}")