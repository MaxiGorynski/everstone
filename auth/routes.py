from flask import Blueprint, render_template, redirect, url_for, flash
from flask_login import logout_user, login_required
from .forms import RegistrationForm, LoginForm
from everstone.models import User

auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    form = RegistrationForm()
    if form.validate_on_submit():
        #Here, add code to save user to the database
        flash('Account created successfully!', 'success')
        return redirect(url_for('auth.login'))
    return render_template('register.html', form=form)

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
    #Here, check user credentials and log them in
        flash('Logged in successfully.', 'success')
        return redirect(url_for('main.dashboard'))
    return render_template('login.html', form=form)

@auth_bp.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have bee logged out', 'info')
    return redirect(url_for('auth.login'))