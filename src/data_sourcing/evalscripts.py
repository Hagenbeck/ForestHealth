from dataclasses import dataclass

from data_sourcing.data_models import EvalScriptType


@dataclass
class RequestType:
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
        "identifier": "default",
        "format": {
            "type": "image/jpeg",
        },
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
        "identifier": "bands",
        "format": {
            "type": "image/tiff",
        },
    },
    {
        "identifier": "scl",
        "format": {
            "type": "image/tiff",
        },
    },
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
  try {
    if (!values || values.length === 0) return 0;
    values.sort((a,b) => a-b);
    var index = Math.floor(values.length / 4);
    return values[index] || 0;
  } catch (e) {
    return 0;
  }
}

function validate(sample) {
  try {
    const invalid = [0, 1, 3, 7, 8, 9, 10];
    return sample && sample.SCL !== undefined && !invalid.includes(sample.SCL);
  } catch (e) {
    return false;
  }
}

function safeDiv(numerator, denominator, defaultValue = 0) {
  if (denominator === 0 || denominator === undefined || denominator === null) {
    return defaultValue;
  }
  let result = numerator / denominator;
  if (isNaN(result) || !isFinite(result)) {
    return defaultValue;
  }
  // Cap extreme values
  return Math.max(-10, Math.min(10, result));
}

function evaluatePixel(samples) {
  try {
    if (!samples || samples.length === 0) {
      return {
        indices: [0, 0, 0, 0, 0, 0, 0, 0, 0],
        scl: [0]
      };
    }

    var valid = samples.filter(validate);
    if (valid.length > 0) {
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

        // All calculations with safe division
        let savi = safeDiv((cl.b08 - cl.b04) * (1 + l), cl.b08 + cl.b04 + l);
        let evi = safeDiv(2.5 * (cl.b08 - cl.b04), cl.b08 + 6 * cl.b04 - 7.5 * cl.b02 + 1);
        let ndre705 = safeDiv(cl.b08 - cl.b05, cl.b08 + cl.b05);
        let ndre740 = safeDiv(cl.b08 - cl.b06, cl.b08 + cl.b06);
        let ndre783 = safeDiv(cl.b08 - cl.b07, cl.b08 + cl.b07);
        let ndvi = safeDiv(cl.b08 - cl.b04, cl.b08 + cl.b04);
        let ndwigao = safeDiv(cl.b08 - cl.b11, cl.b08 + cl.b11);
        let ndwimcf = safeDiv(cl.b03 - cl.b08, cl.b03 + cl.b08);
        let nbr = safeDiv(cl.b08 - cl.b12, cl.b08 + cl.b12);

        // Ensure valid SCL value
        let sclValue = samples[0] && samples[0].SCL !== undefined ? samples[0].SCL : 0;
        sclValue = Math.max(0, Math.min(255, Math.floor(sclValue))); // Ensure valid UINT8

        return {
            indices: [savi, evi, ndre705, ndre740, ndre783, ndvi, ndwigao, ndwimcf, nbr],
            scl: [sclValue]
        };
    }

    return {
        indices: [0, 0, 0, 0, 0, 0, 0, 0, 0],
        scl: [0]
    };
  } catch (e) {
    // Fallback in case of any unexpected error
    return {
        indices: [0, 0, 0, 0, 0, 0, 0, 0, 0],
        scl: [0]
    };
  }
}
"""
indices_response = [
    {
        "identifier": "indices",
        "format": {
            "type": "image/tiff",
        },
    },
    {
        "identifier": "scl",
        "format": {
            "type": "image/tiff",
        },
    },
]

_requests = {
    "RGB": RequestType(rgb_evalscript, rgb_response),
    "ALL": RequestType(all_bands_evalscript, all_bands_response),
    "INDICES": RequestType(indices_evalscript, indices_response),
}


def get_evalscript(mode: EvalScriptType) -> str:
    try:
        return _requests[mode].evalscript
    except KeyError:
        raise KeyError(
            f"{mode} is not a valid Evalscript Type. Valid types are: {list(_requests.keys())}"
        )


def get_response_setup(mode: EvalScriptType) -> list[dict]:
    try:
        return _requests[mode].response
    except KeyError:
        raise KeyError(
            f"{mode} is not a valid Evalscript Type. Valid types are: {list(_requests.keys())}"
        )
