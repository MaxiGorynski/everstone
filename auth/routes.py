from everstone.extensions import bcrypt
from everstone.models import db, User
from flask import Blueprint, render_template, redirect, url_for, flash
from flask_login import logout_user, login_required, login_user
from .forms import RegistrationForm, LoginForm


auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    form = RegistrationForm()
    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
        user = User(username=form.username.data, email=form.email.data, password=hashed_password)
        db.session.add(user)
        db.session.commit()
        login_user(user)
        flash('Account created successfully!', 'success')
        return redirect(url_for('auth.login'))
    return render_template('login.html', form=form)

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    print("Login function has been called.") # Debug, prints if the function is called

    form = LoginForm()
    if form.validate_on_submit():
        print("Form validated successfully.") #Debug
        user = User.query.filter_by(email=form.email.data).first()
        if user:
            print(f"Found user: {user.username}") #Debug, confirm user
        else:
            print("User not found.") #Debug, confirms no user
        try:
            if user and bcrypt.check_password_hash(user.password_hash, form.password.data):
                print(f"User '{user.username}' authenticated successfully.") #Debug
                print("Attempting to log user in...") #Debug before login_user call
                login_user(user, remember=form.remember.data)
                print("User logged in successfully.")
                flash('Logged in successfully.', 'success')
                return redirect(url_for('index')) #Used to be return redirect(url_for('main.index'))
            else:
                print("Authentication failed. Incorrect credentials.")
                flash('Login failed. Please check credentials', 'danger')
        except ValueError as e:
            print(f"ValueError during password check: {e}") #Debug
    else:
        print("Form validation failed.")
        print(form.errors)
    return render_template('login.html', form=form)

@auth_bp.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out', 'info')
    return redirect(url_for('auth.login'))