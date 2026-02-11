"""Galaxy Manager for client-to-galaxy assignments and management.

Handles:
- Client-to-galaxy assignments (round-robin or custom)
- Dynamic reassignments from Layer 5 adaptive re-clustering
- Galaxy-level statistics and health tracking
"""

from typing import Dict, List, Set, Optional
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class GalaxyManager:
    """Manages client assignments to galaxies and handles re-clustering."""
    
    def __init__(self, num_galaxies: int):
        """Initialize galaxy manager.
        
        Args:
            num_galaxies: Total number of galaxies in the system
        """
        self.num_galaxies = num_galaxies
        self.client_to_galaxy: Dict[int, int] = {}
        self.galaxy_to_clients: Dict[int, Set[int]] = {
            gid: set() for gid in range(num_galaxies)
        }
        self.reassignment_history: List[Dict] = []
        
        logger.info(f"GalaxyManager initialized with {num_galaxies} galaxies")
    
    def assign_clients_round_robin(self, client_ids: List[int]) -> Dict[int, int]:
        """Assign clients to galaxies using round-robin strategy.
        
        Args:
            client_ids: List of client IDs to assign
            
        Returns:
            Dict mapping client_id -> galaxy_id
        """
        logger.info(f"Assigning {len(client_ids)} clients to {self.num_galaxies} galaxies (round-robin)")
        
        for idx, client_id in enumerate(client_ids):
            galaxy_id = idx % self.num_galaxies
            self.client_to_galaxy[client_id] = galaxy_id
            self.galaxy_to_clients[galaxy_id].add(client_id)
        
        # Log distribution
        for gid in range(self.num_galaxies):
            logger.info(f"  Galaxy {gid}: {len(self.galaxy_to_clients[gid])} clients")
        
        return self.client_to_galaxy.copy()
    
    def get_galaxy_clients(self, galaxy_id: int) -> List[int]:
        """Get list of clients in a galaxy.
        
        Args:
            galaxy_id: Galaxy ID
            
        Returns:
            List of client IDs in the galaxy
        """
        return list(self.galaxy_to_clients[galaxy_id])
    
    def get_client_galaxy(self, client_id: int) -> Optional[int]:
        """Get galaxy assignment for a client.
        
        Args:
            client_id: Client ID
            
        Returns:
            Galaxy ID or None if client not assigned
        """
        return self.client_to_galaxy.get(client_id)
    
    def reassign_client(self, client_id: int, new_galaxy_id: int, reason: str = "manual"):
        """Reassign a client to a different galaxy.
        
        Args:
            client_id: Client ID to reassign
            new_galaxy_id: New galaxy ID
            reason: Reason for reassignment (e.g., "layer5_reclustering")
        """
        old_galaxy_id = self.client_to_galaxy.get(client_id)
        
        if old_galaxy_id is None:
            logger.warning(f"Client {client_id} not found in any galaxy, assigning to {new_galaxy_id}")
        else:
            # Remove from old galaxy
            self.galaxy_to_clients[old_galaxy_id].discard(client_id)
            logger.info(f"Reassigning client {client_id}: galaxy {old_galaxy_id} -> {new_galaxy_id} (reason: {reason})")
        
        # Add to new galaxy
        self.client_to_galaxy[client_id] = new_galaxy_id
        self.galaxy_to_clients[new_galaxy_id].add(client_id)
        
        # Record history
        self.reassignment_history.append({
            'client_id': client_id,
            'old_galaxy': old_galaxy_id,
            'new_galaxy': new_galaxy_id,
            'reason': reason
        })
    
    def dissolve_galaxy(self, galaxy_id: int, honest_clients: List[int], 
                       malicious_clients: List[int]) -> Dict[int, int]:
        """Dissolve a galaxy and redistribute its clients.
        
        Args:
            galaxy_id: Galaxy to dissolve
            honest_clients: List of honest client IDs to redistribute
            malicious_clients: List of malicious client IDs to quarantine
            
        Returns:
            Dict mapping client_id -> new_galaxy_id for reassigned clients
        """
        logger.warning(f"🔴 DISSOLVING Galaxy {galaxy_id}")
        logger.info(f"  Honest clients: {len(honest_clients)}")
        logger.info(f"  Malicious clients: {len(malicious_clients)} (will be quarantined)")
        
        # Remove all clients from this galaxy
        self.galaxy_to_clients[galaxy_id].clear()
        
        reassignments = {}
        
        # Redistribute honest clients to other galaxies
        other_galaxies = [gid for gid in range(self.num_galaxies) if gid != galaxy_id]
        for idx, client_id in enumerate(honest_clients):
            new_galaxy = other_galaxies[idx % len(other_galaxies)]
            self.reassign_client(client_id, new_galaxy, reason="layer5_dissolution")
            reassignments[client_id] = new_galaxy
            logger.info(f"  ✓ Reassigned honest client {client_id} to galaxy {new_galaxy}")
        
        # Malicious clients are not reassigned (they're quarantined)
        for client_id in malicious_clients:
            if client_id in self.client_to_galaxy:
                del self.client_to_galaxy[client_id]
                logger.info(f"  ✗ Removed malicious client {client_id} from assignments")
        
        logger.info(f"Galaxy {galaxy_id} dissolution complete. Reassigned {len(reassignments)} clients.")
        
        return reassignments
    
    def get_statistics(self) -> Dict:
        """Get galaxy distribution statistics.
        
        Returns:
            Dict with galaxy statistics
        """
        stats = {
            'num_galaxies': self.num_galaxies,
            'total_clients': len(self.client_to_galaxy),
            'galaxy_sizes': {
                gid: len(clients) for gid, clients in self.galaxy_to_clients.items()
            },
            'avg_galaxy_size': len(self.client_to_galaxy) / self.num_galaxies if self.num_galaxies > 0 else 0,
            'reassignments': len(self.reassignment_history)
        }
        
        return stats
    
    def log_status(self):
        """Log current galaxy status."""
        logger.info("=" * 80)
        logger.info("GALAXY STATUS")
        logger.info("=" * 80)
        for gid in range(self.num_galaxies):
            clients = self.galaxy_to_clients[gid]
            logger.info(f"Galaxy {gid:2d}: {len(clients):3d} clients")
        logger.info(f"Total: {len(self.client_to_galaxy)} clients assigned")
        logger.info("=" * 80)
