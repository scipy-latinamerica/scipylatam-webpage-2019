import throttle from 'lodash/throttle'

class Menu {
  constructor () {
    this.setDOM()
    this.setEvents()

    this.states = {
      'OPEN': 'general__menu--open',
      'NOSCROLL': 'body--noscroll',
      'BACKGROUND': 'wrapper-hamburger--scrolled'
    }
  }

  setDOM () {
    this.node = document.querySelector('.general__menu')
    this.button = document.querySelector('.general__hamburger')
    this.wrapperMenu = document.querySelector('.wrapper-hamburger')
    this.body = document.body
  }

  setEvents () {
    this.button.addEventListener('click', this.openMenu.bind(this))
    document.addEventListener('click', this.closeMenu.bind(this))
    window.addEventListener('scroll', throttle(this.setMenuBackground.bind(this), 200))
  }

  openMenu (event) {
    this.node.classList.add(this.states['OPEN'])
    this.body.classList.add(this.states['NOSCROLL'])
  }

  closeMenu (event) {
    if (event.target.classList.contains('container')) {
      this.node.classList.remove(this.states['OPEN'])
      this.body.classList.remove(this.states['NOSCROLL'])
    }
  }

  setMenuBackground () {
    if (window['scrollY'] > 0) {
      this.wrapperMenu.classList.add(this.states['BACKGROUND'])
    } else {
      this.wrapperMenu.classList.remove(this.states['BACKGROUND'])
    }
  }
}

export default Menu
