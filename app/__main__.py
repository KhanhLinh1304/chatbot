from app import app
import app.settings as settings
from controller.route import registryRouter


if __name__ == "__main__":
    registryRouter(app)
    app.run(host=settings.BE_HOST, port=settings.BE_PORT, debug=settings.ENV != "PROD")
