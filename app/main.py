from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.routes.analyze import analyze_bp
from app.config.settings import Config

def create_app():
    static_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'public')
    app = Flask(__name__, static_folder=static_path, static_url_path='')
    
    CORS(app, resources={r"/api/*": {"origins": "*"}}, supports_credentials=True)
    
    @app.route('/')
    def serve_index():
        return send_from_directory(app.static_folder, 'index.html')
    
    @app.route('/api/health')
    def health_check():
        return jsonify({"status": "TrustLens API running"})
    
    app.register_blueprint(analyze_bp, url_prefix='/api/analyze')
    
    @app.errorhandler(404)
    def not_found(e):
        return jsonify({
            "success": False,
            "message": f"Route not found"
        }), 404
    
    @app.errorhandler(Exception)
    def handle_exception(e):
        print(f"Error: {str(e)}")
        return jsonify({
            "success": False,
            "message": str(e)
        }), 500
    
    return app


if __name__ == '__main__':
    app = create_app()
    print(f"🚀 TrustLens backend running on port {Config.PORT}")
    app.run(host='0.0.0.0', port=Config.PORT, debug=True)
