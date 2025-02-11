import '../../assets/header.css'
import Header from '../../components/Header';

function Topheader() {
    const Links = ['Home', 'About', 'Sources', 'Contact']
    return <Header middleText='The solution to your ranked, tournament, or scrimmage games.' Links={Links}></Header>
}
export default Topheader;
