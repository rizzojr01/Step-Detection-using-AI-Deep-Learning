# Step Detection Config API

This API allows you to view and update important step detection parameters at runtime. All endpoints return a single structured object using the `DetectionConfig` Pydantic model.

---

## Endpoints

### 1. Get All Config Values

**GET** `/config`

Returns all important detection config values as a single object.

**Response Example:**

```json
{
  "window_size": 20,
  "start_threshold": 0.5,
  "end_threshold": 0.5,
  "min_step_interval": 10,
  "motion_threshold": 0.2,
  "gyro_threshold": 0.1,
  "min_motion_variance": 0.05,
  "stillness_threshold": 0.01,
  "ai_high_confidence_threshold": 0.8,
  "ai_sensor_disagree_threshold": 0.3
}
```

All fields are optional and may be `null` if not set.

---

### 2. Get Single Config Value

**GET** `/config/{key}`

Returns a config object with only the requested key set, others will be `null`.

**Response Example:**

```json
{
  "window_size": 20,
  "start_threshold": null,
  "end_threshold": null,
  "min_step_interval": null,
  "motion_threshold": null,
  "gyro_threshold": null,
  "min_motion_variance": null,
  "stillness_threshold": null,
  "ai_high_confidence_threshold": null,
  "ai_sensor_disagree_threshold": null
}
```

(Example for `/config/window_size`)

---

### 3. Update Config Value

**POST** `/config/{key}`

Update a config value. Only allowed keys can be updated.

**Request Body:**

```json
{
  "value": "0.6"
}
```

Value should be a string, but will be cast to the correct type automatically.

**Response:**
Returns the full config object after update (same as GET `/config`).

---

## Allowed Keys

- `window_size` (int)
- `start_threshold` (float)
- `end_threshold` (float)
- `min_step_interval` (int)
- `motion_threshold` (float)
- `gyro_threshold` (float)
- `min_motion_variance` (float)
- `stillness_threshold` (float)
- `ai_high_confidence_threshold` (float)
- `ai_sensor_disagree_threshold` (float)

---

## API Model

```python
class DetectionConfig(BaseModel):
    window_size: Optional[int] = None
    start_threshold: Optional[float] = None
    end_threshold: Optional[float] = None
    min_step_interval: Optional[int] = None
    motion_threshold: Optional[float] = None
    gyro_threshold: Optional[float] = None
    min_motion_variance: Optional[float] = None
    stillness_threshold: Optional[float] = None
    ai_high_confidence_threshold: Optional[float] = None
    ai_sensor_disagree_threshold: Optional[float] = None
```

---

## Swagger UI

All endpoints are visible in Swagger UI at `/docs`.

---

## Example Usage

- To get all config values:
  - `GET /config`
- To get a single value:
  - `GET /config/window_size`
- To update a value:
  - `POST /config/start_threshold` with `{ "value": "0.6" }`

---

For further questions or integration help, contact the backend maintainer.
