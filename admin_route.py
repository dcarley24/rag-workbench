import os
import zipfile
from flask import Blueprint, render_template, current_app, redirect, url_for, send_from_directory
from datetime import datetime

admin_bp = Blueprint('admin', __name__)

@admin_bp.route('/admin')
def admin_panel():
    # Get status directly from the KnowledgeBase object
    status = current_app.kb.get_status()
    return render_template('admin.html', status=status)

@admin_bp.route('/admin/reset', methods=['POST'])
def reset_environment():
    # Ask the KnowledgeBase to reset itself
    current_app.kb.reset()
    return redirect(url_for('admin.admin_panel'))

@admin_bp.route('/admin/backup')
def backup_data():
    backup_folder = current_app.config['BACKUP_FOLDER']
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_filename = f"rag_backup_{timestamp}.zip"
    zip_filepath = os.path.join(backup_folder, zip_filename)

    with zipfile.ZipFile(zip_filepath, 'w') as zipf:
        if os.path.exists(current_app.kb.vector_file):
            zipf.write(current_app.kb.vector_file, os.path.basename(current_app.kb.vector_file))
        if os.path.exists(current_app.kb.index_file):
            zipf.write(current_app.kb.index_file, os.path.basename(current_app.kb.index_file))

    return send_from_directory(backup_folder, zip_filename, as_attachment=True)
