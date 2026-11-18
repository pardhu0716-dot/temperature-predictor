"""Add Stage B to the existing Flask application, without altering Stage A files."""
from flask import Blueprint, current_app, jsonify, render_template, request, url_for
from werkzeug.exceptions import BadRequest, RequestEntityTooLarge
from .service import Predictor

web = Blueprint('stage_b_web', __name__, template_folder='templates', static_folder='static', static_url_path='/stage-b/assets')


@web.get('/stage-b')
def page():
    return render_template('stage_b.html')


@web.get('/api/stage-b/model')
def model():
    return jsonify(current_app.extensions['stage_b_predictor'].info)


@web.get('/api/stage-b/floorplans')
def floorplans():
    service = current_app.extensions['stage_b_predictor']
    return jsonify(total_floorplans=len(service.layouts), floorplans=[dict(id=fid, die=service.floorplan(fid)['die']) for fid in service.layouts])


@web.get('/api/stage-b/floorplans/<fid>')
def floorplan(fid):
    try:
        return jsonify(current_app.extensions['stage_b_predictor'].floorplan(fid))
    except KeyError:
        return jsonify(error='Unknown Stage B floorplan'), 404


@web.post('/api/stage-b/predict')
def predict():
    try:
        # Local to this route: Stage A request limits/handling are unchanged.
        if request.content_length is not None and request.content_length > 100_000:
            return jsonify(error='Stage B request exceeds 100 KB'), 413
        raw = request.get_data(cache=True)
        if len(raw) > 100_000:
            return jsonify(error='Stage B request exceeds 100 KB'), 413
        if not request.is_json:
            return jsonify(error='Send application/json'), 415
        payload = request.get_json()
        if not isinstance(payload, dict):
            raise ValueError('Body must be a JSON object')
        service = current_app.extensions['stage_b_predictor']
        if 'batch' in payload:
            batch = payload['batch']
            if not isinstance(batch, list) or not 1 <= len(batch) <= 32:
                raise ValueError('batch must contain 1–32 predictions')
            for p in batch:
                service.parse(p)  # Validate all first; no partial successful responses.
            return jsonify(predictions=[service.predict(p) for p in batch])
        return jsonify(service.predict(payload))
    except (ValueError, TypeError, BadRequest) as exc:
        return jsonify(error=str(exc)), 400
    except KeyError:
        return jsonify(error='Unknown Stage B floorplan'), 404
    except RequestEntityTooLarge:
        return jsonify(error='Stage B request too large'), 413


def register_stage_b(app):
    if 'stage_b_predictor' in app.extensions:
        return app
    app.extensions['stage_b_predictor'] = Predictor()
    app.register_blueprint(web)

    @app.after_request
    def shared_navigation(response):
        if request.path == '/' and response.status_code == 200 and response.mimetype == 'text/html':
            # Add only navigation to the unchanged Stage A HTML response.
            text = response.get_data(as_text=True)
            css = '<link rel="stylesheet" href="' + url_for('stage_b_web.static', filename='stage_b.css') + '">'
            nav = render_template('stage_navigation.html', current_stage='a')
            text = text.replace('</head>', css + '</head>', 1).replace('</header>', '</header>' + nav, 1)
            response.set_data(text)
        return response
    return app
